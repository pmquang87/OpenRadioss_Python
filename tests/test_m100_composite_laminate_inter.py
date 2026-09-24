"""
Tests for Milestone M100: Composite Properties, Laminated Shells & Extended Contact Interfaces Suite
(/PROP/TYPE10, /PROP/TYPE11, /PROP/TYPE16, /PROP/TYPE6, /PLY, /LAMINATE, /INTER/TYPE25, /INTER/SUB).
"""

from __future__ import annotations

import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.model.entities import Ply, Laminate, SubInterface, Property, Interface
from pyradioss.starter.starter import run_starter


_BOILERPLATE = """\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/MAT/LAW1/1
Elastic_Matrix
              7.8e-9
            210000.0                 0.3
/MAT/LAW1/2
Interply_Resin
              1.2e-9
              3500.0                 0.35
/PROP/TYPE1/1
Dummy_Shell
         1         1         1         0         0         0         0
                 1.0                 1.0                 1.0
                 1.0            0.833333
/PART/1
Part_Shell
         1         1
/NODE
         1                 0.0                 0.0                 0.0
         2                 2.0                 0.0                 0.0
         3                 2.0                 2.0                 0.0
         4                 0.0                 2.0                 0.0
/SHELL/1
         1         1         2         3         4
/SURF/SEG/1
Main_Surf
         1         2         3         4
/SURF/SEG/2
Second_Surf
         1         2         3         4
/SKEW/FIX/1
Ref_Skew
                 0.0                 0.0                 0.0
                 1.0                 0.0                 0.0
                 0.0                 1.0                 0.0
"""


def _run(tmp_path, deck, name="TEST_0000.rad"):
    p = tmp_path / name
    p.write_text(deck, encoding="ascii")
    log = MessageLog()
    model = run_starter(str(p), log)
    return model, log


# ══════════════════════════════════════════════════════════════════════
#  /PLY & /LAMINATE tests
# ══════════════════════════════════════════════════════════════════════

class TestPlyAndLaminate:
    """/PLY and /LAMINATE composite stack parsing and validation."""

    def test_ply_and_laminate_fixed(self, tmp_path):
        """Parse fixed-format /PLY and /LAMINATE cards."""
        deck = f"""\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
TEST_PLY_LAMINATE_FIXED
      2021         0
{_BOILERPLATE}
/PLY/1
UD_Carbon_Ply
#   Mat_id               Thick
         1               0.125
/PLY/2
Glass_Woven_Ply
         1                0.25
/LAMINATE/1
Quasi_Isotropic_Stack
#  Ply_id                 Phi                  Zi
        1                 0.0             -0.1875
#Minterply
        2
        2                45.0             -0.0625
        0
        2               -45.0              0.0625
        2
        1                90.0              0.1875
        0
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0

        # Check Plies
        assert 1 in model.plies and 2 in model.plies
        p1 = model.plies[1]
        assert isinstance(p1, Ply)
        assert p1.mat_id == 1
        assert abs(p1.thick - 0.125) < 1e-12

        # Check Laminate
        assert 1 in model.laminates
        lam = model.laminates[1]
        assert isinstance(lam, Laminate)
        assert len(lam.plies) == 4
        assert lam.plies[0].ply_id == 1
        assert abs(lam.plies[0].phi - 0.0) < 1e-12
        assert abs(lam.plies[0].zi - (-0.1875)) < 1e-12
        assert lam.plies[0].mat_interply == 2
        assert lam.plies[1].ply_id == 2
        assert abs(lam.plies[1].phi - 45.0) < 1e-12

    def test_ply_and_laminate_free(self, tmp_path):
        """Parse free-format /PLY and /LAMINATE cards."""
        deck = f"""\
/BEGIN
TEST_PLY_LAMINATE_FREE
{_BOILERPLATE}
/PLY/10
Ply_Ten
1 0.2
/LAMINATE/100
Stack_100
10 30.0 0.1
2
10 -30.0 0.3
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        assert 10 in model.plies
        assert 100 in model.laminates
        lam = model.laminates[100]
        assert len(lam.plies) == 2
        assert lam.plies[0].ply_id == 10
        assert abs(lam.plies[0].phi - 30.0) < 1e-12
        assert lam.plies[0].mat_interply == 2

    def test_laminate_invalid_ref(self, tmp_path):
        """Verify cross-reference error on missing ply."""
        deck = f"""\
/BEGIN
TEST_LAM_ERR
{_BOILERPLATE}
/LAMINATE/1
Error_Stack
999 0.0 0.0
/END
"""
        with pytest.raises(Exception):
            _run(tmp_path, deck)


# ══════════════════════════════════════════════════════════════════════
#  /PROP (TYPE10, TYPE11, TYPE16, TYPE6) tests
# ══════════════════════════════════════════════════════════════════════

class TestCompositeProperties:
    """Composite and orthotropic property cards."""

    def test_prop_sh_comp_fixed(self, tmp_path):
        """Parse fixed-format /PROP/TYPE10 (/PROP/SH_COMP)."""
        deck = f"""\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
TEST_PROP_SH_COMP
      2021         0
{_BOILERPLATE}
/PROP/TYPE10/10
Composite_Shell_Prop
#   Ishell    Ismstr     Ish3n    Idrill                            P_Thick_Fail
        24         4         0         0                                     0.5
#                 Hm                  Hf                  Hr                  Dm                  Dn
                0.01                0.01                0.01                 0.0                 0.0
#        N   Istrain               Thick              Ashear              Ithick     Iplas
         4         1                 2.0            0.833333                   1         1
#                 Vx                  Vy                  Vz     Iskew                            Ip
                 1.0                 0.0                 0.0         1                             0
#              Phi_1               Phi_2               Phi_3               Phi_4
                 0.0                45.0               -45.0                90.0
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        assert 10 in model.properties
        p = model.properties[10]
        assert isinstance(p, Property)
        assert p.type == 10
        assert p.params["ishell"] == 24
        assert abs(p.params["p_thick_fail"] - 0.5) < 1e-12
        assert abs(p.params["thick"] - 2.0) < 1e-12
        assert p.params["skew_id"] == 1
        assert p.params["phi_layers"] == [0.0, 45.0, -45.0, 90.0]

    def test_prop_sh_sandw_fixed(self, tmp_path):
        """Parse fixed-format /PROP/TYPE11 (/PROP/SH_SANDW)."""
        deck = f"""\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
TEST_PROP_SH_SANDW
      2021         0
{_BOILERPLATE}
/PROP/TYPE11/11
Sandwich_Shell_Prop
#   Ishell    Ismstr     Ish3n    Idrill                            P_Thick_Fail
         1         2         0         0                                     0.0
#                 Hm                  Hf                  Hr                  Dm                  Dn
                0.01                0.01                0.01                 0.0                 0.0
#        N   Istrain               Thick              Ashear              Ithick     Iplas
         3         0                 3.0            0.833333                   0         0
#                 Vx                  Vy                  Vz     Iskew     Iorth      Ipos        Ip
                 0.0                 1.0                 0.0         0         1         0         0
#                Phi               Thick                   Z         m                      F_weight
                 0.0                 0.5                -1.0         1                           1.0
                45.0                 2.0                 0.0         2                           1.0
                 0.0                 0.5                 1.0         1                           1.0
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        assert 11 in model.properties
        p = model.properties[11]
        assert p.type == 11
        assert p.params["nip"] == 3
        layers = p.params["layers"]
        assert len(layers) == 3
        assert layers[0]["mat_id"] == 1
        assert abs(layers[0]["thick"] - 0.5) < 1e-12
        assert layers[1]["mat_id"] == 2
        assert abs(layers[1]["phi"] - 45.0) < 1e-12

    def test_prop_sh_fabr_fixed(self, tmp_path):
        """Parse fixed-format /PROP/TYPE16 (/PROP/SH_FABR)."""
        deck = f"""\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
TEST_PROP_SH_FABR
      2021         0
{_BOILERPLATE}
/PROP/TYPE16/16
Fabric_Shell_Prop
#   Ishell    Ismstr     Ish3n                                      P_Thick_Fail
         1         2         0                                               0.2
#                 Hm                  Hf                  Hr                  Dm
                0.01                0.01                0.01                 0.0
#        N   Istrain               Thick              Ashear              Ithick
         2         0                 1.5            0.833333                   1
#                 Vx                  Vy                  Vz   Skew_ID      Ipos                  Ip
                 1.0                 0.0                 0.0         0         0                   0
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        assert 16 in model.properties
        p = model.properties[16]
        assert p.type == 16
        assert abs(p.params["thick"] - 1.5) < 1e-12
        assert abs(p.params["p_thick_fail"] - 0.2) < 1e-12

    def test_prop_sol_orth_fixed(self, tmp_path):
        """Parse fixed-format /PROP/TYPE6 (/PROP/SOL_ORTH)."""
        deck = f"""\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
TEST_PROP_SOL_ORTH
      2021         0
{_BOILERPLATE}
/PROP/TYPE6/6
Solid_Orthotropic_Prop
#   Isolid    Ismstr               Icpre  Itetra10     Inpts   Itetra4    Iframe                  Dn
        14         2                   1         0         0         0         0                 0.0
#                 qa                  qb                   h
                 1.1                0.05                 0.1
#                 Vx                  Vy                  Vz   skew_ID        Ip     Iorth
                 1.0                 0.0                 0.0         1         0         1
#                Phi                  Px                  Py                  Pz
                30.0                 0.0                 0.0                 0.0
#         deltaT_min   Istrain      Ihkt
                 0.0         0         0
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        assert 6 in model.properties
        p = model.properties[6]
        assert p.type == 6
        assert p.params["isolid"] == 14
        assert p.params["skew_id"] == 1
        assert abs(p.params["phi"] - 30.0) < 1e-12

    def test_pcompp_shell_prop_type_ok(self, tmp_path):
        """Assert /PROP/PCOMPP (TYPE51) on /SHELL element parts passes prop_type_ok and build_element_groups."""
        from pyradioss.input.prop_reader import prop_type_ok
        from pyradioss.model.entities import Property

        # Direct prop_type_ok unit check
        prop = Property(id=51, type=51, title="Composite_PCOMPP", params={"laminate_id": 1})
        assert prop_type_ok(1, prop)

        # Starter run with /PART, /SHELL, /PROP/PCOMPP (calling build_element_groups internally)
        deck = f"""\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
TEST_PCOMPP_SHELL
      2021         0
/MAT/LAW1/1
Elastic_Matrix
              7.8e-9
            210000.0                 0.3
/PROP/PCOMPP/51
PCOMPP_Shell_Prop
                   1
/PART/1
Part_Shell
        51         1
/NODE
         1                 0.0                 0.0                 0.0
         2                 2.0                 0.0                 0.0
         3                 2.0                 2.0                 0.0
         4                 0.0                 2.0                 0.0
/SHELL/1
         1         1         2         3         4
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        assert 51 in model.properties
        p = model.properties[51]
        assert p.type == 51
        assert prop_type_ok(1, p)
        assert model.shells is not None
        assert len(model.shells.state["slices"]) == 1


# ══════════════════════════════════════════════════════════════════════
#  /INTER/TYPE25 & /INTER/SUB tests
# ══════════════════════════════════════════════════════════════════════

class TestExtendedInterfaces:
    """/INTER/TYPE25 and /INTER/SUB contact keywords."""

    def test_inter_type25_fixed(self, tmp_path):
        """Parse fixed-format /INTER/TYPE25."""
        deck = f"""\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
TEST_INTER_TYPE25
      2021         0
{_BOILERPLATE}
/INTER/TYPE25/25
Surface_To_Surface_Contact
# surf_ID1  surf_ID2      Istf      Ithe      Igap   Irem_i2                Idel     Iedge
         1         2         4         0         1         1                   0         2
# grnd_IDs                     Gap_scale          %mesh_size           Gap_max_s           Gap_max_m
         0                           1.0                 0.0                 0.5                 2.0
#              Stmin               Stmax     Igap0    Ishape          Edge_angle
                 0.0                 0.0         0         0                 0.0
#              Stfac                Fric                                  Tstart               Tstop
                 1.0                 0.2                                     0.0              1000.0
#      IBC               IVIS2    Inacti                VISs    Ithick                          Pmax
       000                   0         0                 0.0         0                           0.0
#    Ifric    Ifiltr               Xfreq             sens_ID                                 fric_ID
         0         0                 0.0                   0                                       0
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        assert len(model.interfaces) == 1
        itf = model.interfaces[0]
        assert isinstance(itf, Interface)
        assert itf.type == 25
        assert itf.surf_id == 1
        assert itf.surf_id1 == 2
        assert itf.istf == 4
        assert abs(itf.fric - 0.2) < 1e-12
        assert abs(itf.gap - 0.5) < 1e-12
        assert abs(itf.gap_max - 2.0) < 1e-12

    def test_inter_sub_fixed(self, tmp_path):
        """Parse fixed-format /INTER/SUB."""
        deck = f"""\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
TEST_INTER_SUB
      2021         0
{_BOILERPLATE}
/INTER/TYPE7/7
Main_Penalty_Contact
         0         1         4         1
/INTER/SUB/101
Sub_Contact_Pair
# inter_ID  Main_ID1 Second_ID  Main_ID2
         7         1         2         0
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        assert len(model.sub_interfaces) == 1
        sub = model.sub_interfaces[0]
        assert isinstance(sub, SubInterface)
        assert sub.id == 101
        assert sub.inter_id == 7
        assert sub.main_id1 == 1
        assert sub.second_id == 2

    def test_inter_sub_invalid_parent(self, tmp_path):
        """Verify cross-reference error when sub-interface references missing main interface."""
        deck = f"""\
/BEGIN
TEST_INTER_SUB_ERR
{_BOILERPLATE}
/INTER/SUB/101
Sub_Error
999 1 2 0
/END
"""
        with pytest.raises(Exception):
            _run(tmp_path, deck)

    def test_inter_type25_free(self, tmp_path):
        """Parse free-format /INTER/TYPE25."""
        deck = f"""\
/BEGIN
TEST_INTER_TYPE25_FREE
{_BOILERPLATE}
/INTER/TYPE25/25
Surface_To_Surface_Free
1 2 4 1
0 1.0 0.5 2.0
0.0 0.0 0 0 0.0
1.5 0.3 0.0 0.0 500.0
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        assert len(model.interfaces) == 1
        itf = model.interfaces[0]
        assert itf.type == 25
        assert itf.surf_id == 1
        assert itf.surf_id1 == 2
        assert itf.istf == 4
        assert abs(itf.fric - 0.3) < 1e-12
        assert abs(itf.stfac - 1.5) < 1e-12

    def test_inter_sub_free(self, tmp_path):
        """Parse free-format /INTER/SUB."""
        deck = f"""\
/BEGIN
TEST_INTER_SUB_FREE
{_BOILERPLATE}
/INTER/TYPE7/7
Main_Penalty_Contact
0 1 4 1
/INTER/SUB/102
Sub_Contact_Free
7 1 2 0
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        assert len(model.sub_interfaces) == 1
        sub = model.sub_interfaces[0]
        assert sub.id == 102
        assert sub.inter_id == 7
        assert sub.main_id1 == 1
        assert sub.second_id == 2

    def test_composite_props_free(self, tmp_path):
        """Parse free-format /PROP/TYPE10, TYPE11, TYPE16, TYPE6."""
        deck = f"""\
/BEGIN
TEST_PROPS_FREE
{_BOILERPLATE}
/PROP/TYPE10/10
Comp_Free
24 4 0 0 0.5
0.01 0.01 0.01 0.0 0.0
4 1 2.0 0.833333 1 1
1.0 0.0 0.0 1 0
0.0 45.0 -45.0 90.0
/PROP/TYPE11/11
Sandw_Free
1 2 0 0 0.0
0.01 0.01 0.01 0.0 0.0
3 0 3.0 0.833333 0 0
0.0 1.0 0.0 0 1 0 0
0.0 0.5 -1.0 1 1.0
45.0 2.0 0.0 2 1.0
0.0 0.5 1.0 1 1.0
/PROP/TYPE16/16
Fabr_Free
1 2 0 0.2
0.01 0.01 0.01 0.0
2 0 1.5 0.833333 1
1.0 0.0 0.0 0 0 0
/PROP/TYPE6/6
Sol_Orth_Free
14 2 0 1 0 0 0 0 0.0
1.1 0.05 0.1
1.0 0.0 0.0 1 0 1
30.0 0.0 0.0 0.0
0.0 0 0
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        assert 10 in model.properties
        assert 11 in model.properties
        assert 16 in model.properties
        assert 6 in model.properties
        p10 = model.properties[10]
        assert p10.params["phi_layers"] == [0.0, 45.0, -45.0, 90.0]
        p11 = model.properties[11]
        assert len(p11.params["layers"]) == 3
        p16 = model.properties[16]
        assert abs(p16.params["p_thick_fail"] - 0.2) < 1e-12
        p6 = model.properties[6]
        assert abs(p6.params["phi"] - 30.0) < 1e-12

    def test_composite_invalid_refs(self, tmp_path):
        """Verify cross-reference errors on missing skews and materials."""
        deck1 = f"""\
/BEGIN
TEST_ERR1
{_BOILERPLATE}
/PLY/1
Ply_Err
999 0.1
/END
"""
        with pytest.raises(Exception):
            _run(tmp_path, deck1)

        deck2 = f"""\
/BEGIN
TEST_ERR2
{_BOILERPLATE}
/PROP/TYPE10/10
Prop_Skew_Err
24 0 0 0 0.0
0.01 0.01 0.01 0.0 0.0
1 0 1.0 0.833333 0 0
1.0 0.0 0.0 999 0
0.0
/END
"""
        with pytest.raises(Exception):
            _run(tmp_path, deck2)

